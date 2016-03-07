% root = '/home/marsan/workspace/tnative/'
% include(root + 'views/header.tpl')
<table>
  <tr>
    <th>fid</th>
    <th>isad</th>
    <th>title</th>
  </tr>
  % for doc in articles:
    <tr>
      <td><a href="/articles/{{doc.id}}" target='_blank'>{{doc.fid}}</a></td>
      <td>{{doc.isad}}</td>
      <td>{{doc.title}}</td>
    </tr>
  % end 
</table>



<style>
  table, th, td {
    border: 1px solid #ccc;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px 15px;
  }
  td p {
    margin: 2px 5px;
    font-size: 12px;
  }
  tr:nth-child(even) {
    background-color: #f1f1f1;
  }
  b {
    color: darkcyan;
  }
  g {
    color: deeppink; 
    font-weight: bold;
  }
</style>