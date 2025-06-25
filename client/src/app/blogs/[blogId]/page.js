import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
} from "../../../components/ui/table";

export default async function BlogPage({ params }) {
  const res = await fetch("http://localhost:3000/api/blogs", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ id: params.blogId }),
    cache: "no-store", // ensures fresh data for SSR
  });

  const blog = await res.json();

  return (
    <Table>
      <TableCaption>Topics extracted from {blog.title}</TableCaption>
      <TableHeader>
        <TableRow>
          <TableHead>Topic Name</TableHead>
          <TableHead>MCQ IPFS</TableHead>
          <TableHead>Video IPFS</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {blog.topics.map((topic) => (
          <TableRow key={topic.id}>
            <TableCell className="font-medium">{topic.name}</TableCell>
            <TableCell>{topic.mcqIpfsUrl || "Not uploaded"}</TableCell>
            <TableCell>{topic.videoIpfsUrl || "Not uploaded"}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
